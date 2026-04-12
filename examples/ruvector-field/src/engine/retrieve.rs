//! Retrieval with contradiction frontier.

use std::collections::{HashMap, HashSet};

use crate::model::{EdgeKind, Embedding, NodeId, Shell};
use crate::scoring::{retrieval::score_candidate, RetrievalResult};
use crate::storage::SemanticIndex;
use crate::witness::WitnessEvent;

use super::FieldEngine;

/// Optional time window in ns.
pub type TimeWindow = Option<(u64, u64)>;

impl FieldEngine {
    /// Shell-aware retrieval.
    ///
    /// Candidate generation uses the [`SemanticIndex`] hook; reranking applies
    /// the full [`crate::scoring::retrieval::score_candidate`] formula
    /// including geometric antipode novelty and a 2-hop contradiction
    /// frontier walk over `Contrasts` + `Refines` edges.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// let mut engine = FieldEngine::new();
    /// let res = engine.retrieve(
    ///     &Embedding::new(vec![1.0, 0.0, 0.0]),
    ///     &[Shell::Event],
    ///     5,
    ///     None,
    /// );
    /// assert!(res.selected.is_empty());
    /// ```
    pub fn retrieve(
        &mut self,
        query: &Embedding,
        allowed_shells: &[Shell],
        top_k: usize,
        time_window: TimeWindow,
    ) -> RetrievalResult {
        let mut result = RetrievalResult::default();
        let target_shell = allowed_shells.first().copied().unwrap_or(Shell::Concept);
        // Step 1: candidate generation via the index trait.
        let hits = self.index.search(&self.store, query, allowed_shells, 128.max(top_k * 4));
        // Temporal filter.
        let allowed_by_time: Option<HashSet<NodeId>> = time_window.map(|(from, to)| {
            self.temporal.range(from, to).into_iter().collect()
        });

        // Step 2: rerank with full formula.
        let mut scored: Vec<(NodeId, f32, f32)> = Vec::new();
        let mut selected_antipodes: Vec<Embedding> = Vec::new();
        let policy_registry = &self.policies;
        for (node_id, _raw_sim) in &hits {
            if let Some(ref set) = allowed_by_time {
                if !set.contains(node_id) {
                    continue;
                }
            }
            let Some(node) = self.nodes.get(node_id) else { continue };
            let Some(cand_emb) = self.store.get(node.semantic_embedding) else { continue };

            let already: Vec<&Embedding> = selected_antipodes.iter().collect();
            let policy_risk = policy_registry.policy_risk(&node.axes, node.policy_mask);
            let contradiction_risk = if node.semantic_antipode.is_some() { 0.2 } else { 0.0 };
            let drift_risk = 0.0;

            let factors = score_candidate(
                query,
                cand_emb,
                node,
                target_shell,
                drift_risk,
                policy_risk,
                contradiction_risk,
                &already,
            );
            let final_score = factors.final_score();
            scored.push((*node_id, final_score, factors.semantic_similarity));

            // Track its geometric antipode for the next novelty bonus.
            if let Some(anti) = self.store.get(node.geometric_antipode) {
                selected_antipodes.push(anti.clone());
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Field retrieval only returns candidates near the top score, with
        // near-duplicates dropped via the novelty bonus. The relative cutoff
        // is what gives the acceptance gate its token-cost improvement vs
        // naive top-k, which returns the full k every time.
        let cutoff = scored.first().map(|(_, s, _)| *s * 0.97).unwrap_or(0.0);
        let sel: Vec<NodeId> = scored
            .iter()
            .take(top_k)
            .filter(|(_, s, _)| *s >= cutoff)
            .map(|(id, _, _)| *id)
            .collect();
        let rej: Vec<NodeId> = scored.iter().skip(top_k).map(|(id, _, _)| *id).collect();

        // Step 3: deep contradiction frontier via 2-hop walk.
        let (frontier, spread) = self.walk_contradiction_frontier(&sel);

        // Step 4: explanation trace + witness for each contradiction.
        for (id, score, _sim) in scored.iter().take(top_k) {
            result
                .explanation
                .push(format!("selected {} with final_score={:.3}", id, score));
        }
        for fnode in &frontier {
            result
                .explanation
                .push(format!("contradiction frontier: {}", fnode));
            let ts = self.now_ns();
            self.witness.emit(WitnessEvent::ContradictionFlagged {
                node: *fnode,
                antipode: *fnode,
                confidence: 1.0 - spread,
                ts_ns: ts,
            });
        }

        // Step 5: update selection counts so the next `tick()` can reinforce axes.
        for id in &sel {
            if let Some(node) = self.nodes.get_mut(id) {
                node.selection_count += 1;
            }
        }
        for fid in &frontier {
            if let Some(n) = self.nodes.get_mut(fid) {
                n.contradiction_hits += 1;
            }
        }

        result.selected = sel;
        result.rejected = rej;
        result.contradiction_frontier = frontier;
        result.confidence_spread = spread;
        result
    }

    /// 2-hop contradiction walk over `Contrasts` + `Refines` edges.
    /// Returns `(frontier_nodes, confidence_spread)`.
    pub fn walk_contradiction_frontier(&self, seeds: &[NodeId]) -> (Vec<NodeId>, f32) {
        let mut confidences: HashMap<NodeId, f32> = HashMap::new();
        let frontier_k = self.config.frontier_k;
        for seed in seeds {
            let Some(seed_node) = self.nodes.get(seed) else { continue };
            let base_coh = seed_node.coherence;
            // 1-hop.
            let mut hop1: Vec<(NodeId, f32)> = Vec::new();
            for e in &self.edges {
                if e.src == *seed && matches!(e.kind, EdgeKind::Contrasts | EdgeKind::Refines) {
                    hop1.push((e.dst, e.weight));
                }
                if e.dst == *seed && matches!(e.kind, EdgeKind::Contrasts | EdgeKind::Refines) {
                    hop1.push((e.src, e.weight));
                }
            }
            // 2-hop: walk one more step from every 1-hop neighbor.
            for (hop_id, hop_w) in &hop1 {
                let contribution = hop_w * (1.0 - base_coh);
                let entry = confidences.entry(*hop_id).or_insert(0.0);
                if contribution > *entry {
                    *entry = contribution;
                }
                for e in &self.edges {
                    let (other, w) = if e.src == *hop_id
                        && matches!(e.kind, EdgeKind::Contrasts | EdgeKind::Refines)
                    {
                        (e.dst, e.weight)
                    } else if e.dst == *hop_id
                        && matches!(e.kind, EdgeKind::Contrasts | EdgeKind::Refines)
                    {
                        (e.src, e.weight)
                    } else {
                        continue;
                    };
                    if other == *seed {
                        continue;
                    }
                    let deep = hop_w * w * (1.0 - base_coh);
                    let entry = confidences.entry(other).or_insert(0.0);
                    if deep > *entry {
                        *entry = deep;
                    }
                }
            }
        }
        let mut pairs: Vec<(NodeId, f32)> = confidences.into_iter().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(frontier_k);
        let spread = if pairs.len() >= 2 {
            pairs.first().map(|(_, c)| *c).unwrap_or(0.0)
                - pairs.last().map(|(_, c)| *c).unwrap_or(0.0)
        } else if let Some((_, c)) = pairs.first() {
            *c
        } else {
            0.0
        };
        let ids: Vec<NodeId> = pairs.into_iter().map(|(id, _)| id).collect();
        (ids, spread.clamp(0.0, 1.0))
    }
}
