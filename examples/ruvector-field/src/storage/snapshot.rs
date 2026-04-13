//! Field snapshots and diffs — spec section 7.
//!
//! A snapshot captures the aggregate state of the field at a point in time:
//! shell centroids, contradiction frontier, per-shell coherence, drift totals,
//! active routing hints, and witness cursor. Diffs are computed structurally
//! with no external serialization — we ship a plain text format so the demo
//! stays std-only.
//!
//! # Example
//!
//! ```
//! use ruvector_field::storage::FieldSnapshot;
//! let a = FieldSnapshot::default();
//! let b = FieldSnapshot::default();
//! let diff = a.diff(&b);
//! assert!(diff.added_nodes.is_empty());
//! ```

use core::fmt;
use std::collections::{HashMap, HashSet};

use crate::model::{Embedding, HintId, NodeId, Shell, WitnessCursor};
use crate::scoring::{DriftSignal, RoutingHint};

/// Per-shell centroid and coherence summary.
#[derive(Debug, Clone, Default)]
pub struct ShellSummary {
    /// Number of nodes in this shell.
    pub node_count: usize,
    /// Average coherence across this shell.
    pub avg_coherence: f32,
    /// Centroid of the shell's embeddings.
    pub centroid: Vec<f32>,
}

/// Field snapshot.
#[derive(Debug, Clone, Default)]
pub struct FieldSnapshot {
    /// Wall clock of the snapshot.
    pub ts_ns: u64,
    /// Monotonic witness cursor when this snapshot was committed.
    pub witness_cursor: WitnessCursor,
    /// Per-shell summaries keyed by depth.
    pub shell_summaries: [ShellSummary; 4],
    /// Contradiction frontier at snapshot time.
    pub contradiction_frontier: Vec<NodeId>,
    /// Drift totals across all four channels.
    pub drift: DriftSignal,
    /// Active routing hints.
    pub active_hints: Vec<RoutingHint>,
    /// Full node id set for structural diff.
    pub nodes: HashSet<NodeId>,
    /// Edge set represented as `(src, dst, kind_tag)` for Jaccard diffs.
    pub edges: HashSet<(NodeId, NodeId, &'static str)>,
}

impl FieldSnapshot {
    /// Summary for the given shell.
    pub fn summary(&self, shell: Shell) -> &ShellSummary {
        &self.shell_summaries[shell.depth() as usize]
    }

    /// Mutable summary for the given shell.
    pub fn summary_mut(&mut self, shell: Shell) -> &mut ShellSummary {
        &mut self.shell_summaries[shell.depth() as usize]
    }

    /// Compute centroid for one shell from embeddings.
    pub fn fill_centroid<'a, I: IntoIterator<Item = &'a Embedding>>(
        &mut self,
        shell: Shell,
        embeddings: I,
    ) {
        let mut dim = 0usize;
        let mut sum: Vec<f32> = Vec::new();
        let mut count = 0usize;
        for emb in embeddings {
            if sum.is_empty() {
                dim = emb.values.len();
                sum = vec![0.0_f32; dim];
            }
            for (i, v) in emb.values.iter().enumerate().take(dim) {
                sum[i] += v;
            }
            count += 1;
        }
        if count > 0 {
            for v in &mut sum {
                *v /= count as f32;
            }
        }
        let s = self.summary_mut(shell);
        s.centroid = sum;
        s.node_count = count;
    }

    /// Diff against another snapshot.
    pub fn diff(&self, other: &FieldSnapshot) -> SnapshotDiff {
        let added_nodes: Vec<NodeId> = other.nodes.difference(&self.nodes).copied().collect();
        let removed_nodes: Vec<NodeId> = self.nodes.difference(&other.nodes).copied().collect();

        let mut shell_changes: HashMap<Shell, (i64, f32)> = HashMap::new();
        for &s in &Shell::all() {
            let before = self.summary(s);
            let after = other.summary(s);
            let delta_count = after.node_count as i64 - before.node_count as i64;
            let delta_coh = after.avg_coherence - before.avg_coherence;
            if delta_count != 0 || delta_coh.abs() > 1e-4 {
                shell_changes.insert(s, (delta_count, delta_coh));
            }
        }

        let drift_delta = DriftSignal {
            semantic: other.drift.semantic - self.drift.semantic,
            structural: other.drift.structural - self.drift.structural,
            policy: other.drift.policy - self.drift.policy,
            identity: other.drift.identity - self.drift.identity,
            total: other.drift.total - self.drift.total,
        };

        SnapshotDiff {
            added_nodes,
            removed_nodes,
            shell_changes,
            drift_delta,
        }
    }

    /// Render as the simple text format (no serde).
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("snapshot ts_ns={} cursor={}\n", self.ts_ns, self.witness_cursor.0));
        for s in Shell::all() {
            let sum = self.summary(s);
            out.push_str(&format!(
                "  shell {} count={} avg_coh={:.3}\n",
                s, sum.node_count, sum.avg_coherence
            ));
        }
        out.push_str(&format!("  frontier={}\n", self.contradiction_frontier.len()));
        out.push_str(&format!("  drift_total={:.3}\n", self.drift.total));
        out.push_str(&format!("  active_hints={}\n", self.active_hints.len()));
        out
    }

    /// Find an active hint by id.
    pub fn active_hint(&self, id: HintId) -> Option<&RoutingHint> {
        self.active_hints.iter().find(|h| h.id == id)
    }
}

impl fmt::Display for FieldSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

/// Diff between two snapshots.
#[derive(Debug, Clone, Default)]
pub struct SnapshotDiff {
    /// Nodes added between snapshots.
    pub added_nodes: Vec<NodeId>,
    /// Nodes removed between snapshots.
    pub removed_nodes: Vec<NodeId>,
    /// `(delta_count, delta_avg_coherence)` per shell.
    pub shell_changes: HashMap<Shell, (i64, f32)>,
    /// Drift delta across channels.
    pub drift_delta: DriftSignal,
}

impl SnapshotDiff {
    /// `true` if nothing changed.
    pub fn is_empty(&self) -> bool {
        self.added_nodes.is_empty()
            && self.removed_nodes.is_empty()
            && self.shell_changes.is_empty()
            && self.drift_delta.total.abs() < 1e-6
    }
}
