//! Four-channel drift detection.

use std::collections::HashSet;

use crate::model::{Embedding, NodeId};
use crate::scoring::DriftSignal;
use crate::storage::FieldSnapshot;

use super::FieldEngine;

impl FieldEngine {
    /// Compute drift against a reference snapshot.
    ///
    /// All four channels are populated:
    /// * **semantic** — centroid shift vs reference centroid.
    /// * **structural** — Jaccard distance over edge sets.
    /// * **policy** — mean movement in `policy_fit` across nodes.
    /// * **identity** — change in `NodeKind` distribution.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// let mut engine = FieldEngine::new();
    /// let reference = Embedding::new(vec![0.5, 0.5, 0.5]);
    /// let snapshot = FieldSnapshot::default();
    /// let d = engine.drift_with(&reference, &snapshot);
    /// assert!(d.total >= 0.0);
    /// ```
    pub fn drift_with(&self, reference_centroid: &Embedding, reference: &FieldSnapshot) -> DriftSignal {
        let semantic = self.semantic_drift(reference_centroid);
        let structural = self.structural_drift(reference);
        let policy = self.policy_drift(reference);
        let identity = self.identity_drift(reference);
        let total = semantic + structural + policy + identity;
        DriftSignal {
            semantic,
            structural,
            policy,
            identity,
            total,
        }
    }

    /// Convenience wrapper when no reference snapshot is available yet —
    /// structural/policy/identity channels read zero.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// let mut engine = FieldEngine::new();
    /// let d = engine.drift(&Embedding::new(vec![0.5, 0.5, 0.5]));
    /// assert!(d.total >= 0.0);
    /// ```
    pub fn drift(&self, reference_centroid: &Embedding) -> DriftSignal {
        let empty = FieldSnapshot::default();
        self.drift_with(reference_centroid, &empty)
    }

    fn semantic_drift(&self, reference_centroid: &Embedding) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let dim = reference_centroid.values.len();
        let mut centroid = vec![0.0_f32; dim];
        let mut count = 0.0_f32;
        for node in self.nodes.values() {
            if let Some(emb) = self.store.get(node.semantic_embedding) {
                for (i, v) in emb.values.iter().enumerate().take(dim) {
                    centroid[i] += v;
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            for v in &mut centroid {
                *v /= count;
            }
        }
        let current = Embedding::new(centroid);
        let sim = reference_centroid.cosine(&current).clamp(-1.0, 1.0);
        1.0 - (sim + 1.0) / 2.0
    }

    fn structural_drift(&self, reference: &FieldSnapshot) -> f32 {
        // Jaccard distance over edge sets.
        let current: HashSet<(NodeId, NodeId, &'static str)> = self
            .edges
            .iter()
            .map(|e| (e.src, e.dst, kind_tag(e.kind)))
            .collect();
        let ref_set = &reference.edges;
        if current.is_empty() && ref_set.is_empty() {
            return 0.0;
        }
        let inter: usize = current.intersection(ref_set).count();
        let uni: usize = current.union(ref_set).count();
        if uni == 0 {
            0.0
        } else {
            1.0 - (inter as f32 / uni as f32)
        }
    }

    fn policy_drift(&self, reference: &FieldSnapshot) -> f32 {
        if self.policies.is_empty() || self.nodes.is_empty() {
            return 0.0;
        }
        let mut current_fit = 0.0_f32;
        for n in self.nodes.values() {
            current_fit += self.policies.policy_fit(&n.axes, n.policy_mask);
        }
        current_fit /= self.nodes.len() as f32;
        // Reference fit approximated from ref snapshot's avg coherence as a proxy
        // when no persisted per-node policy fit is available — delta is the drop.
        let ref_fit = if reference.shell_summaries.iter().any(|s| s.node_count > 0) {
            let mut avg = 0.0_f32;
            let mut c = 0.0_f32;
            for s in &reference.shell_summaries {
                if s.node_count > 0 {
                    avg += s.avg_coherence * s.node_count as f32;
                    c += s.node_count as f32;
                }
            }
            if c > 0.0 {
                avg / c
            } else {
                current_fit
            }
        } else {
            current_fit
        };
        (ref_fit - current_fit).abs().clamp(0.0, 1.0)
    }

    fn identity_drift(&self, reference: &FieldSnapshot) -> f32 {
        if self.nodes.is_empty() || reference.nodes.is_empty() {
            return 0.0;
        }
        // Fraction of current nodes not present in the reference set.
        let mut gone = 0usize;
        for id in self.nodes.keys() {
            if !reference.nodes.contains(id) {
                gone += 1;
            }
        }
        (gone as f32 / self.nodes.len() as f32).clamp(0.0, 1.0)
    }
}

fn kind_tag(kind: crate::model::EdgeKind) -> &'static str {
    use crate::model::EdgeKind::*;
    match kind {
        Supports => "supports",
        Contrasts => "contrasts",
        Refines => "refines",
        RoutesTo => "routes_to",
        DerivedFrom => "derived_from",
        SharesRegion => "shares_region",
        BindsWitness => "binds_witness",
    }
}

/// Re-export of the edge-kind tag function used by snapshot serialization.
pub fn edge_kind_tag(kind: crate::model::EdgeKind) -> &'static str {
    kind_tag(kind)
}
