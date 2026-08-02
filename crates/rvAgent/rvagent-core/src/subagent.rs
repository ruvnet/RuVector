//! Subagent boundary — a tool that spawns an isolated context and returns a
//! String (ADR-275).
//!
//! Deliberately *not* peer agents with a message bus, shared mutable state, or
//! a mergeable state type. One writer, auxiliary intelligence around it, never
//! parallel writes.
//!
//! The evidence for coding tasks is one-directional: at equal token budget,
//! single-agent matches or beats multi-agent on interdependent work, and the
//! team that shipped a parallel-writer architecture walked it back after a year
//! of production data. A CRDT can merge two edits to the same file without
//! textual conflict; it cannot make the *result* coherent. That is the failure
//! this boundary is shaped to make unrepresentable — a subagent that can only
//! return a `String` cannot write, so there is nothing to merge.

use async_trait::async_trait;

use crate::error::Result;

/// What a subagent is for.
///
/// Both adopted roles are read-only. The distinction is what context they get,
/// which is load-bearing rather than cosmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentRole {
    /// Evaluates an artifact with **no inherited conversation** — only the diff
    /// and the task statement.
    ///
    /// The lack of shared context is the point, not an optimization: reviewers
    /// measurably perform *better* without the parent's history, because a
    /// shorter window means less context rot and none of the parent's
    /// accumulated rationalizations.
    ///
    /// **Gated, not adopted** (ADR-278 §7). metaharness ADR-226 is a
    /// gold-scored null on a closely related design — a read-only strong
    /// advisor produced zero marginal resolves at 5.4× cost. That advisor saw
    /// the full transcript where this reviewer sees only the diff, so it is not
    /// refuted, but this role must demonstrate marginal lift over a
    /// no-reviewer control before it reaches the default path.
    Reviewer,
    /// Explores, reads, greps; returns a summary string.
    ///
    /// Keeps exploration output out of the main window entirely, which is the
    /// cleanest lever on wasted-context accumulation — the earliest and most
    /// universal long-run failure.
    Gatherer,
}

impl SubagentRole {
    /// Whether this role may see the parent's conversation.
    ///
    /// Always false. Encoded as a method rather than assumed, so adding a role
    /// that inherits context is a visible decision rather than an oversight.
    pub fn inherits_parent_context(&self) -> bool {
        false
    }

    /// Whether this role may use state-mutating tools.
    ///
    /// Always false — that is what makes this a single-writer architecture.
    pub fn may_write(&self) -> bool {
        false
    }

    /// The model tier this role should run on.
    ///
    /// The gatherer runs cheap on measured grounds: a specialized small model
    /// matched a frontier-mini in that slot, and putting a frontier model there
    /// bought +0.4 pp at 5.8× cost — corroborated independently by ADR-226's
    /// 5.4× null. Model tiering per role is part of the design, not a later
    /// optimization.
    pub fn model_tier(&self) -> ModelTier {
        match self {
            SubagentRole::Reviewer => ModelTier::Standard,
            SubagentRole::Gatherer => ModelTier::Cheap,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SubagentRole::Reviewer => "reviewer",
            SubagentRole::Gatherer => "gatherer",
        }
    }
}

/// Which model tier a subagent runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    Cheap,
    Standard,
}

/// A subagent invocation: everything it gets, and nothing more.
///
/// There is no parent-state field by construction. A subagent cannot reach the
/// parent's conversation, files, or todos even by accident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentRequest {
    pub role: SubagentRole,
    /// The complete input. For a reviewer this is the diff plus the task
    /// statement; for a gatherer, the question to answer.
    pub prompt: String,
}

impl SubagentRequest {
    pub fn new(role: SubagentRole, prompt: impl Into<String>) -> Self {
        Self {
            role,
            prompt: prompt.into(),
        }
    }
}

/// The boundary itself: isolated context in, String out.
///
/// The signature is the architecture. A subagent returns text and nothing else
/// — no state update, no file handle, no mergeable value — so a caller cannot
/// wire one up as a concurrent writer even if it wanted to.
#[async_trait]
pub trait Subagent: Send + Sync {
    async fn run(&self, request: SubagentRequest) -> Result<String>;
}

/// Cap on a subagent's returned summary, in bytes.
///
/// A subagent exists to *reduce* what reaches the parent's context. One that
/// returns its entire transcript has inverted its own purpose, so the boundary
/// enforces the budget rather than trusting the callee.
pub const MAX_SUMMARY_BYTES: usize = 8_000;

/// Truncate a subagent's return value to the summary budget.
pub fn enforce_summary_budget(summary: String) -> String {
    crate::masking::truncate_tool_result(summary, MAX_SUMMARY_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoSubagent;

    #[async_trait]
    impl Subagent for EchoSubagent {
        async fn run(&self, request: SubagentRequest) -> Result<String> {
            Ok(format!("{}: {}", request.role.as_str(), request.prompt))
        }
    }

    #[test]
    fn no_role_may_write() {
        // The single-writer invariant. If this ever fails, the architecture
        // has changed and ADR-275 needs revisiting first.
        for role in [SubagentRole::Reviewer, SubagentRole::Gatherer] {
            assert!(!role.may_write(), "{} must not write", role.as_str());
        }
    }

    #[test]
    fn no_role_inherits_parent_context() {
        for role in [SubagentRole::Reviewer, SubagentRole::Gatherer] {
            assert!(!role.inherits_parent_context());
        }
    }

    #[test]
    fn gatherer_runs_on_the_cheap_tier() {
        // A frontier model in this slot measured +0.4 pp at 5.8x cost.
        assert_eq!(SubagentRole::Gatherer.model_tier(), ModelTier::Cheap);
    }

    #[tokio::test]
    async fn boundary_returns_only_a_string() {
        let out = EchoSubagent
            .run(SubagentRequest::new(SubagentRole::Gatherer, "where is X?"))
            .await
            .unwrap();
        assert_eq!(out, "gatherer: where is X?");
    }

    #[test]
    fn request_carries_no_parent_state() {
        // Compile-time property, asserted structurally: the request is exactly
        // a role and a prompt. Adding a parent-state field would break this.
        let r = SubagentRequest::new(SubagentRole::Reviewer, "diff");
        assert_eq!(r.role, SubagentRole::Reviewer);
        assert_eq!(r.prompt, "diff");
    }

    #[test]
    fn oversized_summaries_are_capped() {
        let huge = "x".repeat(MAX_SUMMARY_BYTES * 2);
        let capped = enforce_summary_budget(huge);
        assert!(capped.len() <= MAX_SUMMARY_BYTES);
        assert!(capped.ends_with("[output truncated]"));
    }

    #[test]
    fn short_summaries_pass_through_unchanged() {
        let s = "found it in src/lib.rs".to_string();
        assert_eq!(enforce_summary_budget(s.clone()), s);
    }
}
