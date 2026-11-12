//! # ruvector-staged-workspace
//!
//! Content-hash + revision-id state binding for RuV artifacts, implementing
//! the **StagedWorkspace (arXiv:2608.18050)** pattern as a RuV invariant
//! per **ADR-318** (`docs/adr/ADR-318-stagedworkspace-content-hash-state-binding.md`,
//! PIR WP16, ruvnet/RuVector#863).
//!
//! ## The invariant (ADR-318 §Decision)
//!
//! 1. **Every artifact gets a content hash and a revision id at write time.**
//!    The hash is SHA-256 over the artifact's canonical byte representation,
//!    computed with the repo's existing RVF-layer implementation
//!    ([`rvf_types::sha256`]) — the same SHA-256 the program's canonical
//!    receipt contract (ruflo ADR-322C, adopted via ADR-312) specifies.
//! 2. **Every downstream reference binds to that exact hash**, not just the
//!    artifact's identity: a [`StagedView`] records the [`ArtifactRef`]
//!    (hash + revision) of the exact version it was derived from, for all
//!    five view kinds ADR-318 enumerates ([`ViewKind`]: parser result, tool
//!    call, diff, approval, output).
//! 3. **Stale state is automatically invalid.** When the workspace's live
//!    hash for an artifact no longer matches a view's recorded hash,
//!    [`WorkspaceState::validate`] and [`WorkspaceState::execute`] reject
//!    the view with a hard error — a structural, fail-closed rejection at
//!    read time, never a soft warning.
//! 4. Workspace-state transitions are anchored through the **ADR-312 seam**
//!    ([`TransitionAnchor`]): each commit produces a
//!    [`WorkspaceTransitionRecord`] with a canonical byte encoding, a
//!    SHA-256 record id (mirroring ADR-322C's `receiptId = SHA-256(canonical
//!    payload)` identity derivation), and a domain-separated signing
//!    preimage (`domain || 0x00 || canonicalBytes`, ADR-322C's scheme).
//!    Per the "no anchor, no transition" discipline (the same fail-closed
//!    posture as ADR-134's "no witness, no mutation" in
//!    `ruvector-agent-memory`), an anchor rejection aborts the commit.
//!
//! ## Preprint-reproduction rule (read before citing numbers)
//!
//! StagedWorkspace (arXiv:2608.18050) reports +8.3–12.1pp OfficeQA Pass@1
//! for dual parsed/native access on the paper's own models and reference
//! implementation, **which is not publicly released** ("Under Review", no
//! repo — verified in `06-wave2-evidence-review.md` §5). This crate is a
//! first-party implementation from the paper's description. **The paper's
//! figure is NOT claimed here**: per ADR-318 §Decision 5 and ADR-306,
//! promotion requires this program's own `research-gate`-recomputed delta on
//! an internal OfficeQA-equivalent task set — a future benchmark deliverable,
//! not part of this slice.

pub mod adapter;
pub mod anchor;
pub mod artifact;
pub mod view;
pub mod workspace;

pub use anchor::{
    AnchorError, AnchorReceipt, EvidenceGrade, NoopAnchor, RejectingAnchor, TransitionAnchor,
    WorkspaceTransitionRecord, TRANSITION_DOMAIN,
};
pub use artifact::{ArtifactRef, ContentHash};
pub use view::{StagedView, ViewKind};
pub use workspace::{
    LineageSnapshot, ViewConflict, WorkspaceError, WorkspaceSnapshot, WorkspaceState,
};
