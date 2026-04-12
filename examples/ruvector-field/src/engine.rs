//! Minimal field engine: builder, scoring, retrieval, drift, routing.
//!
//! Everything here is intentionally simple — no ANN, no HNSW, no async —
//! so the shape of the spec stays visible. Swap in real backends when
//! promoting this to a production crate.

use std::collections::HashMap;

use crate::types::*;

pub struct FieldEngine {
    pub nodes: HashMap<u64, FieldNode>,
    pub edges: Vec<FieldEdge>,
    next_id: u64,
}

impl FieldEngine {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_id: 1,
        }
    }

    /// Ingest a raw interaction. Everything lands in the `Event` shell
    /// initially. Promotion happens later via `promote_candidates`.
    pub fn ingest(
        &mut self,
        kind: NodeKind,
        text: impl Into<String>,
        embedding: Embedding,
        axes: AxisScores,
        policy_mask: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let geometric_antipode = embedding.geometric_antipode();
        let node = FieldNode {
            id,
            kind,
            semantic_embedding: embedding,
            geometric_antipode,
            semantic_antipode: None,
            shell: Shell::Event,
            axes,
            coherence: 0.5,
            continuity: 0.5,
            resonance: 0.0,
            policy_mask,
            witness_ref: None,
            ts_ns: now_ns(),
            text: text.into(),
        };
        let resonance = Self::resonance_score(&node);
        let mut node = node;
        node.resonance = resonance;
        self.nodes.insert(id, node);
        id
    }

    /// Explicit semantic antipode. Spec section 10.2 — this is the link that
    /// powers contradiction reasoning, not a vector flip.
    pub fn bind_semantic_antipode(&mut self, a: u64, b: u64, weight: f32) {
        if let Some(na) = self.nodes.get_mut(&a) {
            na.semantic_antipode = Some(b);
        }
        if let Some(nb) = self.nodes.get_mut(&b) {
            nb.semantic_antipode = Some(a);
        }
        let ts = now_ns();
        self.edges.push(FieldEdge {
            src: a,
            dst: b,
            kind: EdgeKind::Contrasts,
            weight,
            ts_ns: ts,
        });
        self.edges.push(FieldEdge {
            src: b,
            dst: a,
            kind: EdgeKind::Contrasts,
            weight,
            ts_ns: ts,
        });
    }

    pub fn add_edge(&mut self, src: u64, dst: u64, kind: EdgeKind, weight: f32) {
        self.edges.push(FieldEdge {
            src,
            dst,
            kind,
            weight,
            ts_ns: now_ns(),
        });
    }

    /// Spec section 8.1: multiplicative resonance, bounded to [0, 1].
    fn resonance_score(node: &FieldNode) -> f32 {
        node.axes.product() * node.coherence.clamp(0.0, 1.0) * node.continuity.clamp(0.0, 1.0)
    }

    /// Spec section 8.2: coherence from effective resistance proxy.
    /// Here we approximate "effective resistance" by the inverse of the
    /// average cosine similarity to the same-shell neighborhood.
    pub fn recompute_coherence(&mut self) {
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        for id in ids {
            let node = self.nodes[&id].clone();
            let mut sims = Vec::new();
            for (oid, other) in &self.nodes {
                if *oid == id || other.shell != node.shell {
                    continue;
                }
                sims.push(node.semantic_embedding.cosine(&other.semantic_embedding));
            }
            let avg_sim = if sims.is_empty() {
                0.0
            } else {
                sims.iter().sum::<f32>() / sims.len() as f32
            };
            // Map similarity (-1..1) into an effective resistance proxy.
            let eff_resistance = (1.0 - avg_sim).max(0.0);
            let coherence = 1.0 / (1.0 + eff_resistance);
            let n = self.nodes.get_mut(&id).unwrap();
            n.coherence = coherence;
            n.resonance = Self::resonance_score(n);
        }
    }

    /// Spec section 9.1: promotion with minimal hysteresis.
    ///
    /// - recurrence is counted as edges of kind `DerivedFrom` or `Supports`.
    /// - contradiction risk is approximated by the number of `Contrasts` edges.
    pub fn promote_candidates(&mut self) -> Vec<(u64, Shell, Shell)> {
        let mut promotions = Vec::new();
        let support: HashMap<u64, usize> = self.count_edges(&[EdgeKind::Supports, EdgeKind::DerivedFrom]);
        let contrast: HashMap<u64, usize> = self.count_edges(&[EdgeKind::Contrasts]);

        for (id, node) in self.nodes.iter_mut() {
            let s = *support.get(id).unwrap_or(&0);
            let c = *contrast.get(id).unwrap_or(&0);
            let before = node.shell;

            let after = match node.shell {
                Shell::Event if s >= 2 && node.resonance > 0.15 => Shell::Pattern,
                Shell::Pattern if s >= 3 && c == 0 && node.coherence > 0.65 => Shell::Concept,
                Shell::Concept if s >= 4 && c == 0 && node.resonance > 0.35 => Shell::Principle,
                other => other,
            };
            if after != before {
                node.shell = after;
                promotions.push((*id, before, after));
            }
        }
        promotions
    }

    fn count_edges(&self, kinds: &[EdgeKind]) -> HashMap<u64, usize> {
        let mut out = HashMap::new();
        for e in &self.edges {
            if kinds.contains(&e.kind) {
                *out.entry(e.dst).or_insert(0) += 1;
                *out.entry(e.src).or_insert(0) += 1;
            }
        }
        out
    }

    /// Spec section 11: shell aware retrieval with contradiction frontier.
    pub fn retrieve(
        &self,
        query: &Embedding,
        allowed_shells: &[Shell],
        top_k: usize,
    ) -> RetrievalResult {
        let mut scored: Vec<(u64, f32)> = Vec::new();
        let mut explanation: Vec<String> = Vec::new();

        for node in self.nodes.values() {
            if !allowed_shells.contains(&node.shell) {
                continue;
            }
            let sim = query.cosine(&node.semantic_embedding).clamp(-1.0, 1.0);
            let semantic_similarity = (sim + 1.0) / 2.0; // to [0,1]

            let shell_fit = 1.0 - (node.shell.depth() as f32 * 0.05);
            let coherence_fit = node.coherence;
            let continuity_fit = node.continuity;
            let resonance_fit = 0.5 + 0.5 * node.resonance;

            let candidate_score = semantic_similarity
                * shell_fit
                * coherence_fit
                * continuity_fit
                * resonance_fit;

            // risk
            let contradiction_risk = if node.semantic_antipode.is_some() { 0.15 } else { 0.0 };
            let drift_risk = 0.0;
            let policy_risk = 0.0;
            let risk = contradiction_risk + drift_risk + policy_risk;
            let safety = 1.0 / (1.0 + risk);

            let final_score = candidate_score * safety;
            scored.push((node.id, final_score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<u64> = scored.iter().take(top_k).map(|(id, _)| *id).collect();
        let rejected: Vec<u64> = scored.iter().skip(top_k).map(|(id, _)| *id).collect();

        // Contradiction frontier: semantic antipodes of selected nodes.
        let mut frontier = Vec::new();
        for id in &selected {
            if let Some(n) = self.nodes.get(id) {
                if let Some(anti) = n.semantic_antipode {
                    frontier.push(anti);
                    explanation.push(format!(
                        "node {} has semantic antipode {} — flagged on contradiction frontier",
                        id, anti
                    ));
                }
            }
        }
        for (id, score) in scored.iter().take(top_k) {
            explanation.push(format!("selected node {} with final_score={:.3}", id, score));
        }

        RetrievalResult {
            selected,
            rejected,
            contradiction_frontier: frontier,
            explanation,
        }
    }

    /// Spec section 12: drift across four channels. Alert when total > threshold
    /// and at least two channels agree. We return the raw signal; the caller
    /// decides whether to alert.
    pub fn drift(&self, reference_centroid: &Embedding) -> DriftSignal {
        if self.nodes.is_empty() {
            return DriftSignal::default();
        }
        let dim = reference_centroid.values.len();
        let mut centroid = vec![0.0_f32; dim];
        for n in self.nodes.values() {
            for (i, v) in n.semantic_embedding.values.iter().enumerate() {
                centroid[i] += v;
            }
        }
        let count = self.nodes.len() as f32;
        for v in &mut centroid {
            *v /= count;
        }
        let centroid = Embedding::new(centroid);
        let sim = reference_centroid.cosine(&centroid).clamp(-1.0, 1.0);
        let semantic = 1.0 - (sim + 1.0) / 2.0;

        let structural = (self.edges.len() as f32 / (count + 1.0)).min(1.0) * 0.1;
        let policy = 0.0;
        let identity = 0.0;
        let total = semantic + structural + policy + identity;

        DriftSignal {
            semantic,
            structural,
            policy,
            identity,
            total,
        }
    }

    /// Spec section 13: routing hint issuance. Cheap eligibility only —
    /// the hint itself is advisory and carries a TTL.
    pub fn route(&self, query: &Embedding, roles: &[(u64, &str, Embedding)]) -> Option<RoutingHint> {
        let mut best: Option<(u64, &str, f32)> = None;
        for (agent_id, role, role_embed) in roles {
            let capability_fit = (query.cosine(role_embed) + 1.0) / 2.0;
            let locality_fit = 0.9;
            let shell_fit = 0.9;
            let role_fit = 1.0;
            let expected_gain = 0.6;
            let expected_cost = 0.2_f32.max(0.01);
            let score = capability_fit
                * role_fit
                * locality_fit
                * shell_fit
                * (expected_gain / expected_cost);
            if best.map(|(_, _, s)| score > s).unwrap_or(true) {
                best = Some((*agent_id, role, score));
            }
        }
        best.map(|(agent_id, role, score)| RoutingHint {
            target_partition: None,
            target_agent: Some(agent_id),
            gain_estimate: score.min(10.0) / 10.0,
            cost_estimate: 0.2,
            ttl_epochs: 4,
            reason: format!("best role match: {}", role),
        })
    }
}
