//! Routing and snapshots.

use crate::error::FieldError;
use crate::model::{Embedding, HintId, NodeId, Shell};
use crate::policy::PolicyRegistry;
use crate::scoring::routing::{score_route, RouteInputs};
use crate::scoring::RoutingHint;
use crate::storage::FieldSnapshot;
use crate::witness::WitnessEvent;

use super::drift::edge_kind_tag;
use super::FieldEngine;

/// Agent descriptor for routing.
#[derive(Debug, Clone)]
pub struct RoutingAgent {
    /// Agent id.
    pub agent_id: u64,
    /// Role name.
    pub role: String,
    /// Capability embedding.
    pub capability: Embedding,
    /// Role embedding (task distribution centroid).
    pub role_embedding: Embedding,
    /// Representative partition node (for BFS distance).
    pub home_node: Option<NodeId>,
    /// Shell this agent naturally operates at.
    pub home_shell: Shell,
}

impl FieldEngine {
    /// Register a [`PolicyRegistry`].
    pub fn set_policy_registry(&mut self, registry: PolicyRegistry) {
        self.policies = registry;
    }

    /// Compute the best routing hint among `agents`.
    ///
    /// Unlike the demo's old hardcoded version, every factor — capability fit,
    /// role fit, locality fit (BFS partition distance), shell fit, expected
    /// gain, and expected cost — is derived from live engine state.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// use ruvector_field::engine::route::RoutingAgent;
    /// let mut engine = FieldEngine::new();
    /// let q = Embedding::new(vec![1.0, 0.0, 0.0]);
    /// let agents = vec![RoutingAgent {
    ///     agent_id: 1,
    ///     role: "verifier".into(),
    ///     capability: Embedding::new(vec![0.9, 0.1, 0.0]),
    ///     role_embedding: Embedding::new(vec![0.8, 0.1, 0.1]),
    ///     home_node: None,
    ///     home_shell: Shell::Concept,
    /// }];
    /// let hint = engine.route(&q, Shell::Concept, &agents, None, false).unwrap();
    /// assert_eq!(hint.target_agent, Some(1));
    /// ```
    pub fn route(
        &mut self,
        query: &Embedding,
        target_shell: Shell,
        agents: &[RoutingAgent],
        query_node: Option<NodeId>,
        requires_proof: bool,
    ) -> Option<RoutingHint> {
        let mut best: Option<(RoutingHint, f32)> = None;
        for agent in agents {
            let partition_distance = match (query_node, agent.home_node) {
                (Some(q), Some(h)) => self.partition_distance(q, h),
                _ => 1,
            };
            let expected_gain = 0.3 + 0.7 * query.cosine01(&agent.capability);
            let expected_cost = 0.1 + 0.05 * partition_distance as f32
                + (self.nodes.len() as f32 / 1_000.0).min(0.5);

            let factors = score_route(&RouteInputs {
                query,
                capability: &agent.capability,
                role: &agent.role_embedding,
                partition_distance,
                agent_shell: agent.home_shell,
                target_shell,
                expected_gain,
                expected_cost,
            });
            let id = self.next_hint_id();
            let hint = RoutingHint {
                id,
                target_partition: None,
                target_agent: Some(agent.agent_id),
                target_shell: Some(target_shell),
                capability_fit: factors.capability_fit,
                role_fit: factors.role_fit,
                locality_fit: factors.locality_fit,
                shell_fit: factors.shell_fit,
                gain_estimate: expected_gain,
                cost_estimate: expected_cost,
                ttl_epochs: 4,
                requires_proof,
                committed: false,
                reason: format!("best role match: {}", agent.role),
            };
            let score = factors.product();
            let better = match &best {
                Some((_, s)) => score > *s,
                None => true,
            };
            if better {
                best = Some((hint, score));
            }
        }
        let (hint, _) = best?;
        let ts = self.now_ns();
        self.active_hints.insert(hint.id, hint.clone());
        self.witness
            .emit(WitnessEvent::RoutingHintIssued { hint: hint.id, ts_ns: ts });
        Some(hint)
    }

    /// Commit an active hint through a proof gate. Emits `RoutingHintCommitted`.
    pub fn commit_hint<G: crate::proof::ProofGate>(
        &mut self,
        id: HintId,
        gate: &mut G,
    ) -> Result<(), FieldError> {
        let hint = self
            .active_hints
            .get_mut(&id)
            .ok_or(FieldError::UnknownEdge(id.0))?;
        match hint.commit(gate) {
            Ok(_) => {
                let ts = self.now_ns();
                self.witness
                    .emit(WitnessEvent::RoutingHintCommitted { hint: id, ts_ns: ts });
                Ok(())
            }
            Err(crate::proof::ProofError::Denied(why)) => Err(FieldError::ProofDenied(why)),
            Err(crate::proof::ProofError::NotRequired) => Err(FieldError::ProofRequired),
        }
    }

    /// Capture a snapshot of the current field state.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// let mut engine = FieldEngine::new();
    /// let snap = engine.snapshot();
    /// assert!(snap.nodes.is_empty());
    /// ```
    pub fn snapshot(&mut self) -> FieldSnapshot {
        let mut snap = FieldSnapshot::default();
        let ts = self.now_ns();
        snap.ts_ns = ts;
        snap.witness_cursor = self.witness.cursor();

        for s in Shell::all() {
            let embs: Vec<&Embedding> = self
                .nodes
                .values()
                .filter(|n| n.shell == s)
                .filter_map(|n| self.store.get(n.semantic_embedding))
                .collect();
            let node_count = embs.len();
            let mut avg_coh = 0.0;
            for n in self.nodes.values().filter(|n| n.shell == s) {
                avg_coh += n.coherence;
            }
            if node_count > 0 {
                avg_coh /= node_count as f32;
            }
            snap.fill_centroid(s, embs.into_iter());
            let summary = snap.summary_mut(s);
            summary.avg_coherence = avg_coh;
        }

        snap.nodes = self.nodes.keys().copied().collect();
        snap.edges = self
            .edges
            .iter()
            .map(|e| (e.src, e.dst, edge_kind_tag(e.kind)))
            .collect();
        snap.active_hints = self.active_hints.values().cloned().collect();
        snap.contradiction_frontier = {
            let seeds: Vec<NodeId> = self
                .nodes
                .values()
                .filter(|n| n.semantic_antipode.is_some())
                .map(|n| n.id)
                .take(8)
                .collect();
            self.walk_contradiction_frontier(&seeds).0
        };
        self.witness
            .emit(WitnessEvent::FieldSnapshotCommitted { cursor: snap.witness_cursor, ts_ns: ts });
        snap
    }
}
