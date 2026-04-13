//! Promotion and demotion with hysteresis.
//!
//! Spec sections 9.1 and 9.2. A node must satisfy promotion criteria across
//! `config.promotion_passes` consecutive calls to [`FieldEngine::promote_candidates`]
//! and spend at least `config.min_residence_ns` in its current shell before it
//! can move. Demotion fires on support decay, contradiction growth, persistent
//! drift, or oscillation inside `config.hysteresis_window`.

use core::fmt;

use crate::model::{EdgeKind, NodeId, Shell};
use crate::witness::WitnessEvent;

use super::FieldEngine;

/// Why a promotion fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionReason {
    /// Event → Pattern: recurrence + resonance window held.
    RecurrenceThreshold,
    /// Pattern → Concept: compression + low contradiction.
    CompressionStable,
    /// Concept → Principle: high resonance + zero contradictions.
    PolicyInvariant,
    /// Demotion: support decayed below threshold.
    SupportDecay,
    /// Demotion: contradictions climbed.
    ContradictionGrowth,
    /// Demotion: oscillation inside hysteresis window.
    Oscillation,
}

impl fmt::Display for PromotionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PromotionReason::RecurrenceThreshold => f.write_str("recurrence threshold"),
            PromotionReason::CompressionStable => f.write_str("compression stable"),
            PromotionReason::PolicyInvariant => f.write_str("policy invariant"),
            PromotionReason::SupportDecay => f.write_str("support decay"),
            PromotionReason::ContradictionGrowth => f.write_str("contradiction growth"),
            PromotionReason::Oscillation => f.write_str("oscillation inside hysteresis window"),
        }
    }
}

/// One promotion / demotion record.
#[derive(Debug, Clone)]
pub struct PromotionRecord {
    /// Node affected.
    pub node: NodeId,
    /// Shell before the transition.
    pub from: Shell,
    /// Shell after the transition.
    pub to: Shell,
    /// Which rule fired.
    pub reason: PromotionReason,
}

impl fmt::Display for PromotionRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}→{} ({})", self.node, self.from, self.to, self.reason)
    }
}

impl FieldEngine {
    /// Run one promotion pass.
    ///
    /// Nodes that cross a promotion threshold bump their `promotion_streak`;
    /// only when the streak reaches `config.promotion_passes` and the node has
    /// been in its shell for at least `config.min_residence_ns` do they move.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// let mut engine = FieldEngine::new();
    /// let out = engine.promote_candidates();
    /// assert!(out.is_empty());
    /// ```
    pub fn promote_candidates(&mut self) -> Vec<PromotionRecord> {
        let support = self.count_edges(&[EdgeKind::Supports, EdgeKind::DerivedFrom]);
        let contrast = self.count_edges(&[EdgeKind::Contrasts]);
        let mut records: Vec<PromotionRecord> = Vec::new();
        let now = self.now_ns();
        let passes_required = self.config.promotion_passes;
        let residence = self.config.min_residence_ns;
        let hysteresis = self.config.hysteresis_window;

        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for id in ids {
            let s = *support.get(&id).unwrap_or(&0);
            let c = *contrast.get(&id).unwrap_or(&0);
            let mut upsert_req: Option<(crate::model::EmbeddingId, Shell)> = None;
            let mut promoted: Option<(Shell, Shell, PromotionReason)> = None;
            {
                let Some(node) = self.nodes.get_mut(&id) else { continue };

                let (candidate, reason) = match node.shell {
                    Shell::Event if s >= 2 && node.resonance > 0.12 => {
                        (Some(Shell::Pattern), PromotionReason::RecurrenceThreshold)
                    }
                    Shell::Pattern if s >= 3 && c == 0 && node.coherence > 0.55 => {
                        (Some(Shell::Concept), PromotionReason::CompressionStable)
                    }
                    Shell::Concept if s >= 4 && c == 0 && node.resonance > 0.25 => {
                        (Some(Shell::Principle), PromotionReason::PolicyInvariant)
                    }
                    _ => (None, PromotionReason::RecurrenceThreshold),
                };

                if let Some(target) = candidate {
                    node.promotion_streak += 1;
                    let residence_ok = now.saturating_sub(node.shell_entered_ts) >= residence;
                    let history_window: Vec<Shell> = node
                        .promotion_history
                        .iter()
                        .rev()
                        .take(hysteresis)
                        .copied()
                        .collect();
                    let oscillating = history_window
                        .windows(2)
                        .any(|w| w[0] == target || w[1] == node.shell);
                    if node.promotion_streak >= passes_required && residence_ok && !oscillating {
                        let before = node.shell;
                        node.shell = target;
                        node.shell_entered_ts = now;
                        node.promotion_streak = 0;
                        node.promotion_history.push(target);
                        if node.promotion_history.len() > hysteresis * 2 {
                            let drop = node.promotion_history.len() - hysteresis * 2;
                            node.promotion_history.drain(0..drop);
                        }
                        upsert_req = Some((node.semantic_embedding, target));
                        promoted = Some((before, target, reason));
                    }
                } else {
                    node.promotion_streak = 0;
                }
            }
            if let Some((eid, target)) = upsert_req {
                self.index_upsert(id, eid, target);
            }
            if let Some((before, target, reason)) = promoted {
                records.push(PromotionRecord {
                    node: id,
                    from: before,
                    to: target,
                    reason,
                });
                self.witness.emit(WitnessEvent::ShellPromoted {
                    node: id,
                    from: before,
                    to: target,
                    ts_ns: now,
                });
            }
        }
        records
    }

    /// Run one demotion pass.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// let mut engine = FieldEngine::new();
    /// assert!(engine.demote_candidates().is_empty());
    /// ```
    pub fn demote_candidates(&mut self) -> Vec<PromotionRecord> {
        let support = self.count_edges(&[EdgeKind::Supports, EdgeKind::DerivedFrom]);
        let contrast = self.count_edges(&[EdgeKind::Contrasts]);
        let mut records: Vec<PromotionRecord> = Vec::new();
        let now = self.now_ns();

        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for id in ids {
            let s = *support.get(&id).unwrap_or(&0);
            let c = *contrast.get(&id).unwrap_or(&0);
            let mut upsert_req: Option<(crate::model::EmbeddingId, Shell)> = None;
            let mut demoted: Option<(Shell, Shell, PromotionReason)> = None;
            {
                let Some(node) = self.nodes.get_mut(&id) else { continue };

                let (need, reason) = match node.shell {
                    Shell::Pattern if s < 1 => (true, PromotionReason::SupportDecay),
                    Shell::Concept if s < 2 || c >= 2 => {
                        if c >= 2 {
                            (true, PromotionReason::ContradictionGrowth)
                        } else {
                            (true, PromotionReason::SupportDecay)
                        }
                    }
                    Shell::Principle if c >= 1 => (true, PromotionReason::ContradictionGrowth),
                    _ => (false, PromotionReason::SupportDecay),
                };

                if need {
                    if let Some(target) = node.shell.demote() {
                        let before = node.shell;
                        node.shell = target;
                        node.shell_entered_ts = now;
                        node.promotion_streak = 0;
                        node.promotion_history.push(target);
                        upsert_req = Some((node.semantic_embedding, target));
                        demoted = Some((before, target, reason));
                    }
                }
            }
            if let Some((eid, target)) = upsert_req {
                self.index_upsert(id, eid, target);
            }
            if let Some((before, target, reason)) = demoted {
                records.push(PromotionRecord {
                    node: id,
                    from: before,
                    to: target,
                    reason,
                });
                self.witness.emit(WitnessEvent::ShellDemoted {
                    node: id,
                    from: before,
                    to: target,
                    ts_ns: now,
                });
            }
        }
        records
    }
}
